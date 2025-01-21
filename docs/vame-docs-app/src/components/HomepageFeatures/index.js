import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Behavioral Segmentation',
    description: (
      <>
        VAME provides advanced algorithms for precise behavioral segmentation, allowing researchers to analyze animal motion patterns efficiently.
      </>
    ),
  },
  {
    title: 'Machine Learning Framework',
    description: (
      <>
        Utilizing state-of-the-art machine learning techniques, VAME extracts meaningful insights from behavioral data to facilitate scientific discoveries.
      </>
    ),
  },
  {
    title: 'Python API',
    description: (
      <>
        VAME offers a straight forward Python API, making it easy for users to integrate into their workflows and perform behavioral analysis with minimal effort.
      </>
    ),
  },
];

function Feature({ Svg, title, description }) {
  return (
    <div className={clsx('col col--4')}>
      {/* <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div> */}
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
